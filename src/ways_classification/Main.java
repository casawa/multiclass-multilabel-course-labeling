/* Class: Main.java
 * ------------------
 * This class collects the description and WAYS from ExploreCourses
 * of classes that satisfy at least one WAY.
 */

package ways_classification;

import edu.stanford.services.explorecourses.ExploreCoursesConnection;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Collection;
import java.util.List;
import java.util.Set;

import org.jdom2.JDOMException;

import edu.stanford.services.explorecourses.Course;
import edu.stanford.services.explorecourses.School;
import edu.stanford.services.explorecourses.Department;

public class Main {

	public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
		System.out.println("Starting");
		ExploreCoursesConnection connection = new ExploreCoursesConnection();
		System.out.println("Got Connection");

		PrintWriter writer = new PrintWriter("new_courses.txt", "UTF-8");

		int num_courses = 0;
		int num_satisfy_way = 0;
		int num_ways = 0;
		try {
			Set<School> schools = connection.getSchools();
			for(School school : schools) {
				List<Department> depts = school.getDepartments();
				for(Department dept : depts) {
					List<Course> courses = connection.getCoursesByQuery(dept.getCode());
					for(Course course : courses) {
						num_courses += 1;
						Collection<String> satisfies = course.getGeneralEducationRequirementsSatisfied();
						Boolean satisfies_way = false;
						for(String satisfy : satisfies) {
							if(satisfy.startsWith("WAY")) {
								writer.println(satisfy);
								satisfies_way = true;
								num_ways += 1;
							}
						}
						if(satisfies_way) {
							writer.println(course.getAcademicOrganization() + " " + course.getSubjectCodePrefix() + " " + course.getDescription());
							satisfies_way = false;
							num_satisfy_way += 1;
						}
					}
					
				}
			}	
		} catch (IOException e) {
			// Catch IO exceptions
			e.printStackTrace();
		} catch (JDOMException e) {
			// Catch JDOM exceptions
			e.printStackTrace();
		}
		
		writer.close();
		System.out.print(num_courses);
		System.out.print('\n');
		System.out.print(num_satisfy_way);
		System.out.print('\n');
		System.out.print(num_ways);
	}

}
