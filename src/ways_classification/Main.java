/* Class: Main.java
 * ------------------
 * This class collects the description and WAYS from ExploreCourses
 * of classes that satisfy at least one WAY.
 */

package ways_classification;

import edu.stanford.services.explorecourses.ExploreCoursesConnection;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Set;

import org.jdom2.JDOMException;

import edu.stanford.services.explorecourses.Course;
import edu.stanford.services.explorecourses.School;
import edu.stanford.services.explorecourses.Department;

public class Main {

	public static void main(String[] args) {
		System.out.println("Starting");
		ExploreCoursesConnection connection = new ExploreCoursesConnection();
		System.out.println("Got Connection");

		try {
			Set<School> schools = connection.getSchools();
			for(School school : schools) {
				List<Department> depts = school.getDepartments();
				for(Department dept : depts) {
					List<Course> courses = connection.getCoursesByQuery(dept.getCode());
					for(Course course : courses) {
						Collection<String> satisfies = course.getGeneralEducationRequirementsSatisfied();
						Boolean satisfies_way = false;
						for(String satisfy : satisfies) {
							if(satisfy.startsWith("WAY")) {
								System.out.println(satisfy);
								satisfies_way = true;
							}
						}
						if(satisfies_way) {
							System.out.println(course.getDescription());
							satisfies_way = false;
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
		
	}

}
