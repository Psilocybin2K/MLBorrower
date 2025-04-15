using HandlebarsDotNet;

namespace MLBorrowerProfile.Utils
{
    public static class HandlebarsUtility
    {
        /// <summary>
        /// Compiles and renders a Handlebars template using a JSON payload.
        /// </summary>
        /// <param name="templateContent">The raw Handlebars template as a string.</param>
        /// <param name="jsonPayload">A valid JSON string matching the template's expected data structure.</param>
        /// <returns>The rendered result string.</returns>
        public static string RenderTemplate(string templateContent, object payload)
        {
            HandlebarsTemplate<object, object> template = Handlebars.Compile(templateContent);

            string renderedResult = template(payload);

            return renderedResult;
        }
    }
}